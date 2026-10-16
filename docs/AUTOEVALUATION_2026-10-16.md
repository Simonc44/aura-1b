# Autoevaluation chiffree du depot — 16 oct. 2026

**Note : 77 500 / 100 000** (~78 %).

Ce n'est pas une note d'ecole et ce n'est pas un standard externe : la grille est
ecrite ici, en 7 axes, et **chaque point perdu a une raison mesurable**. Les
mesures, elles, ne sont pas negociables : elles sont listees dans le tableau
« Ce qui a ete mesure sur ce HEAD », avec la commande qui les produit. La note
est discutable ; les chiffres ne le sont pas.

## La grille

| Axe | Maximum | Obtenu | Ce qui coute des points |
|---|---|---|---|
| A. Exactitude de la documentation | 15 000 | **11 500** | des comptes ecrits a la main (−1 500), un « pourquoi » a une seule cause sur deux (−1 000), des docs publiques non relues apres un changement de code (−1 000) |
| B. Filet de tests | 20 000 | **15 000** | 28 % du code jamais execute par la suite (−2 500), aucun plancher de couverture (−1 500), 3 tests qui ne mesuraient rien pendant des semaines (−1 000) |
| C. Typage et lint | 10 000 | **9 000** | `mypy aura/` seulement : `tests/` (11 104 lignes) et `scripts/` (9 880) hors typage (−1 000) |
| D. Securite des dependances | 12 000 | **8 500** | un `.venv` local a pu deriver sans que rien ne le dise (−1 500), 1 alerte Dependabot non instruite (−1 500), `pip-audit -r` inutilisable sous Windows (−500) |
| E. CI et reproductibilite | 13 000 | **9 000** | la CI est **ROUGE** sur `main` depuis au moins le 11 oct., 5 jobs sur 11 (−2 000), dont un `mypy` qui echoue **en silence** (−1 000), 2 jobs non executables depuis ce poste (−1 000) |
| F. Architecture et lisibilite | 15 000 | **11 500** | 74 modules pour un 1B (−1 500), couverture tres inegale entre modules (−1 000), l'elagage du plus gros module est chiffre mais non fait (−1 000) |
| G. Mesures et honnetete des benchmarks | 15 000 | **13 000** | pas de banc externe complet (−1 000), aucun classement defendable face aux 3B/8B (−1 000) |
| **Total** | **100 000** | **77 500** | |

## Ce qui a ete mesure sur ce HEAD

| Mesure | Valeur | D'ou elle vient |
|---|---|---|
| Tests collectes / passes / **sautes** | 1159 / 1159 / **0** | `.venv/Scripts/python.exe -m pytest -q` — 69,6 s |
| Couverture de `aura/` + `rontogpt/` | **72 %** : 9 535 instructions, 2 563 non couvertes, 460 branches partielles | `pytest -q --cov` |
| Typage | **0 erreur sur 74 fichiers** (corps non annotes compris) | `.venv/Scripts/python.exe -m mypy aura/` |
| Lint | **0 violation** | `uvx ruff check .` |
| Typage de la TUI | `tsc --noEmit` → **0** (1 448 lignes TypeScript) | `cd tui && npx tsc --noEmit` |
| Plancher Python annonce | **3.11.16** : `compileall aura rontogpt tests scripts` → exit 0 | `/c/Users/admin/.local/bin/python3.11.exe -m compileall -q …` |
| Audit des dependances | verrou : **1 avis accepte et ecrit** (`diskcache`, aucun correctif publie) ; venv local : **3 avis `urllib3`** (2.7.0 installe contre 2.8.0 fige) | `python scripts/audit_dependances.py` |
| Verrou | 172 paquets figes, audites | `uv.lock` |
| Taille du code | 41 755 lignes Python (`aura` 20 656, `tests` 11 104, `scripts` 9 880), 1 448 TS, 3 558 Markdown, 298 fichiers suivis | `git ls-files … \| xargs wc -l` |
| CI | **8 jobs** : suite Ubuntu+Windows × 3.11/3.12, chaine `.aef`, typage, installeur, lint, compat-311, securite, TUI | `.github/workflows/tests.yml` |
| CI sur `main` | **ROUGE** : les 4 jobs de tests **et** `mypy` echouent, sur les **30 derniers runs** (depuis au moins le 11 oct. 2026) | API GitHub `/actions/workflows/tests.yml/runs` |
| Jobs de CI **rejoues ici** | **5** : suite (Windows, 3.12 seul), typage, lint, compat-311, TUI — plus l'**audit** du job `securite`, qui a parle (voir D) | les commandes ci-dessus |
| Dependabot | **1 alerte moderee, non instruite** | message de `git push` (GitHub) |

## Detail par axe

### A. Exactitude de la documentation — 11 500 / 15 000

Le depot a une culture rare de la correction : `§3nonies` a corrige une
couverture annoncee a **20 %** (fausse, a cause d'un `sys.settrace(None)` dans
`aura/potcode.py`), `§3octies` un depot qui **ne compilait pas** sous le Python
annonce, et `§3decies` trois tests qui se sautaient partout. Le 16 oct. 2026,
le chantier de mesures a fait tomber quatre chiffres perimes (1152 → 1157 →
1159 tests, 9 527 → 9 535 instructions) et une cause incomplete (le « chemin
trop long » de `pip-audit` n'etait que la **seconde** des deux erreurs).

Ce qui coute : les comptes sont ecrits **a la main** dans les docs et faux a
chaque changement ; certains « pourquoi » s'arretent a la premiere cause ; des
documents grand public (`README`, `MODEL_CARD`) ont garde du perime apres un
changement de code.

### B. Filet de tests — 15 000 / 20 000

1 159 tests, **0 saute sur une machine complete**, 66 fichiers, ordre aleatoire
(`pytest-randomly`), suite en **70-79 s**, et 11 104 lignes de tests pour
20 656 lignes de `aura/` (ratio 0,54).

Ce qui coute : **28 % du code n'est jamais execute** par la suite (dont
`aura/tui.py` a 24 %) ; il n'y a **aucun plancher** de couverture, donc une
baisse ne rougit pas ; et trois tests ont pu ne rien mesurer pendant des
semaines sans que personne ne le voie — le vrai defaut, corrige le 16 oct.

### C. Typage et lint — 9 000 / 10 000

`mypy` traverse les corps de fonctions non annotes et rend **0 erreur sur
74 fichiers** ; `ruff` 0 ; `tsc --noEmit` 0 sur la TUI ; le plancher 3.11 est
**verifie par un vrai interpreteur**, pas par une cible de configuration.

Ce qui coute : le typage s'arrete a `aura/`. `tests/` et `scripts/`
(20 984 lignes a eux deux) n'en beneficient pas.

### D. Securite des dependances — 8 500 / 12 000

172 paquets figes et audites automatiquement a chaque poussee, **1 avis accepte
et ecrit noir sur blanc** (`diskcache`, aucune version corrective publiee) plutot
que tolere en silence.

Ce qui coute : un `.venv` local a pu **deriver du verrou** (urllib3 2.7.0 contre
2.8.0 fige) et annoncer trois avis qui n'existaient pas dans le verrou — l'outil
qui le detecte (`scripts/audit_dependances.py`) n'existe que depuis le 16 oct. ;
**une alerte Dependabot moderee reste non instruite** ; et `pip-audit --no-deps
-r` ne peut pas aboutir sous Windows (deux causes, contournement documente).

### E. CI et reproductibilite — 9 000 / 13 000

Huit jobs, une matrice Ubuntu + Windows × 3.11/3.12, les dependances tirees des
**metadonnees du projet** (`uv sync --frozen`) et jamais d'une liste recopiee.
**Cinq jobs ont ete rejoues ici** (suite Windows 3.12, typage, lint,
compat-311, TUI), et l'audit du job `securite` a ete execute — il a trouve
l'ecart du venv au lieu d'un vert de complaisance.

**Correction du 16 oct. 2026, et elle coute 1 500 points.** Cette note avait
d'abord ecrit que « le vert GitHub reste une inference » : c'etait faux, et
verifiable en une requete. Les **30 derniers runs de `Tests` sont rouges**,
depuis au moins le 11 oct. 2026, y compris sur des commits anterieurs a ce
chantier. Cinq jobs sur onze tombent : les **quatre entrees de la matrice**
(Ubuntu + Windows × 3.11/3.12) et **`mypy aura/`**.

La cause des tests est etablie, et elle se lit **sans jeton** : le workflow
transforme chaque echec en annotation. **Sept tests exigent le cerveau local**
et assertent un nom derive de son nom de fichier ; sans `modeles/*.gguf` ils
comparent `'inconnu-1b'` a `'-1b'`, et `aura.llama` journalise « GGUF
introuvable ». La documentation affirmait qu'ils « se sautent » : c'est faux,
ils **echouent** — la meme classe de defaut que §3decies, un test dont le sujet
est l'etat du disque.

La cause de `mypy` n'est **pas** identifiee : ce job n'annote rien, et l'API
GitHub n'expose pas les journaux sans jeton. Ce qui est etabli : meme version
(mypy 2.3.1, celle du verrou), meme configuration, **0 erreur sur 74 fichiers**
en local, cache incremental vide ou non — et le job tourne sous Linux, alors
qu'il passe ici sous Windows. Une etape qui annote les erreurs a ete ajoutee
pour que le prochain run dise pourquoi, au lieu de rester muet trois semaines
de plus.

Ce qui coute encore : la chaine `.aef` sous Linux et l'installeur PowerShell
**n'ont pas ete executes** depuis ce poste, et la suite sous 3.11 n'a pas ete
rejouee aujourd'hui (sa derniere mesure date de `§3septies`).

### F. Architecture et lisibilite — 11 500 / 15 000

Un **registre d'experts declaratif** qui porte l'invariant (« aucun approximatif
avant un exact ») et que les tests comparent a l'ordre **reellement execute**,
des contrats chiffres (`aura/contrats.py`), des decoupages recents menes sans
changer la surface publique (`post_traitement`, bus de flux), et 941 lignes
d'`ARCHITECTURE.md` qui documentent les **refus** autant que les choix.

Ce qui coute : 74 modules et 20 656 lignes pour un modele de 1B, c'est une
surface large ; la couverture est **tres inegale** selon les modules ; et
l'elagage du plus gros module est ecrit, chiffre, et **pas fait**.

### G. Mesures et honnetete des benchmarks — 13 000 / 15 000

Banc interne **94/100** ; GSM8K cerveau nu **0,16 / 0,20**, ramene a **0,12**
par le pipeline — l'ecart est publie, pas masque ; les taches a choix multiple
sont **refusees par le code** plutot que transformees en score de hasard ; et le
16 oct., une sonde d'interpretabilite a ete publiee **avec ses reserves**
(prompt sonde sans gabarit de conversation, 4B et non le 1B livre).

Ce qui coute : **aucun banc externe complet** (lm-eval reste partiel) et
**aucun classement defendable** face a un 3B ou un 8B — ce qui est admis, mais
reste une zone aveugle du dossier de preuves.

## Ce que cette note ne vaut pas

- La **grille est mienne** : changer les poids change la note. Les mesures, non.
- L'axe CI est **corrige** (voir E) : la CI n'etait pas seulement « inferee »,
  elle est **rouge**, et c'est une requete qui l'a dit. Le trou de `mypy` reste
  ouvert : son journal n'est pas lisible depuis ce poste.
- **Rien n'est verifie sur les chemins Linux** (`.aef`, installeur PowerShell).
- L'alerte **Dependabot n'a pas ete lue** : elle peut valoir une deduction de
  plus, ou aucune.
- Une note globale **cache des ecarts** : 72 % de couverture en moyenne peut
  vouloir dire 90 % ici et 24 % la — ce qui est justement le cas.

## Reproduire chaque chiffre

    .venv/Scripts/python.exe -m pytest -q              # 1159 passes, 0 saute
    .venv/Scripts/python.exe -m pytest -q --cov        # 72 %, 9 535 instructions
    .venv/Scripts/python.exe -m mypy aura/             # 0 erreur, 74 fichiers
    uvx ruff check .                                   # 0 violation
    cd tui && npx tsc --noEmit                         # 0
    /c/Users/admin/.local/bin/python3.11.exe -m compileall -q aura rontogpt tests scripts
    python scripts/audit_dependances.py                 # avis du verrou (et ecart du venv)
    curl -s https://api.github.com/repos/Simonc44/rontogpt/actions/workflows/tests.yml/runs  # 30 runs, tous rouges
