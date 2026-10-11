"""Pont vers ``lm-evaluation-harness`` : noter le CERVEAU BRUT d'RontoGPT.

Deux mesures, deux objets -- ne pas les confondre :

  * ``scripts/harnais_evaluation.py`` mesure le SYSTEME (le pipeline RontoGPT :
    niveau 0, RAG, graphe de faits, solveurs) sur ses VRAIS points d'entree ;
  * CE script mesure le CERVEAU SEUL (le GGUF), comme on note un modele public
    -- utile pour situer RontoGPT et verifier qu'un re-entrainement des
    adaptateurs ne degrade pas la base.

Il ne recharge PAS le modele : il tape l'API OpenAI-compatible du serveur
llama.cpp deja lance par RontoGPT. Le serveur doit donc tourner.

Prerequis (une seule fois) -- le venv n'a pas de ``pip``, passer par uv :
    uv pip install --python .venv/Scripts/python.exe lm-eval

Taches a LOGLIKELIHOOD : impossibles contre llama.cpp
-----------------------------------------------------
llama.cpp (verifie sur b7400) ignore ``echo: true`` : ``/v1/completions`` ne
renvoie les logprobs que des tokens GENERES, jamais ceux du PROMPT. Or toute
tache a choix multiple de lm-eval (arc_*, hellaswag, mmlu, winogrande, piqa,
boolq...) se note en decoupant les logprobs du prompt, via
``logprobs["token_logprobs"][ctxlen:-1]``. Contre ce serveur la mesure est donc
soit une erreur bruyante, soit -- pire -- un score de HASARD silencieux (cf.
llama.cpp issue #27174). Ce script REFUSE ces taches : mieux vaut pas de
chiffre qu'un chiffre faux.

Ce qui se mesure vraiment : les taches GENERATIVES (gsm8k, triviaqa, ...), ou
le modele ecrit sa reponse et ou lm-eval compare le TEXTE. Aucun logprob n'est
necessaire.

Usage :
    python scripts/lm_eval_aura.py --taches gsm8k --limite 100
    python scripts/lm_eval_aura.py --taches gsm8k --limite 100 --brut
    python scripts/lm_eval_aura.py --taches gsm8k --fewshot 8 --sortie resultats/
    python scripts/lm_eval_aura.py --liste        # taches disponibles (info)
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import sysconfig
from pathlib import Path

#: Adresse du serveur llama.cpp (API OpenAI-compatible). Meme variable que
#: aura/serveur_lora.py, pour rester aligne sur le reste du projet.
_ADRESSE_DEFAUT = "127.0.0.1:8757"

#: Tache par defaut : generative, donc mesurable (cf. en-tete du module).
_TACHES_DEFAUT = "gsm8k"

#: Types de sortie qui se notent en lisant les logprobs des tokens du PROMPT.
#: ``multiple_choice`` emprunte exactement le meme chemin que ``loglikelihood``
#: (une passe par choix, score = somme des logprobs du prompt) : refuse aussi.
# Bornes de la resolution des YAML (groupes -> sous-taches -> tag ->
# include) : mmlu descend de 4 niveaux avant d'atteindre le template.
_PROFONDEUR_MAX = 6

# lm-eval imprime un tableau qui contient « haut » (higher_is_better). Sur une
# console Windows en cp1252, l'enfant meurt en UnicodeEncodeError APRES avoir
# ecrit ses resultats : on perd le rapport et on croit que le run a echoue.
# On force donc l'UTF-8 chez l'enfant, sans dependre de l'environnement ambiant.
_ENV_UTF8 = {**os.environ, "PYTHONIOENCODING": "utf-8"}

_TYPES_LOGPROBS = ("loglikelihood", "loglikelihood_rolling",
                   "multiple_choice", "multiple_choice_cot")


def adresse() -> str:
    """Adresse du serveur (AURA_SERVEUR_ADRESSE, sinon defaut)."""
    return os.environ.get("AURA_SERVEUR_ADRESSE", _ADRESSE_DEFAUT)


def base_url(adresse_serveur: str | None = None, chat: bool = False) -> str:
    """URL OpenAI-compatible attendue par lm-eval.

    ``chat=True`` vise ``/v1/chat/completions`` : c'est le serveur qui applique
    alors le gabarit de conversation du GGUF (ce qu'il faut pour un modele
    instruct comme Gemma-3-1B-it). ``chat=False`` vise le complet brut.
    """
    route = "chat/completions" if chat else "completions"
    return f"http://{adresse_serveur or adresse()}/v1/{route}"


def dossier_taches() -> Path:
    """Repertoire ``tasks`` de lm-eval, localise SANS importer lm_eval.

    On lit les YAML de configuration directement : aucune dependance lourde
    (ni datasets, ni torch) n'est chargee pour savoir comment une tache se note.
    """
    return Path(sysconfig.get_paths()["purelib"]) / "lm_eval" / "tasks"


def _champs(chemin: Path) -> dict[str, str]:
    """Les couples ``cle: valeur`` d'un YAML de tache, SANS regex.

    Ces fichiers sont plats (une cle par ligne). Lire ligne a ligne evite
    toute question d'echappement, et se relit sans effort.
    """
    champs: dict[str, str] = {}
    try:
        contenu = chemin.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return champs
    for ligne in contenu.splitlines():
        nu = ligne.strip()
        if not nu or nu.startswith("#") or ":" not in nu:
            continue
        cle, _, valeur = nu.partition(":")
        champs.setdefault(cle.strip().strip('"'), valeur.strip().strip('"'))
    return champs


def _sous_taches(chemin: Path) -> list[str]:
    """Les noms listes sous ``task:`` — uniquement la forme en LISTE.

    ``task: mmlu_abstract_algebra`` (une seule ligne) n'est pas une liste :
    c'est le nom de la tache elle-meme, et le prendre pour un enfant ferait
    tourner la resolution en rond.
    """
    noms: list[str] = []
    dedans = False
    try:
        contenu = chemin.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return noms
    for ligne in contenu.splitlines():
        nu = ligne.strip()
        if nu.startswith("task:"):
            dedans = not nu[len("task:"):].strip()
            continue
        if not dedans:
            continue
        if nu.startswith("-"):
            noms.append(nu[1:].strip().strip('"'))
        elif nu and not nu.startswith("#"):
            break
    return noms


_YAMLS: list[Path] | None = None
_INDEX_NOMS: dict[str, list[Path]] | None = None


def _yamls(racine: Path | None = None) -> list[Path]:
    """YAML de taches, sous ``racine`` ou partout.

    Sans ``racine``, tout lm-eval est parcouru UNE SEULE fois et garde : le
    refaire par appel coutait une minute (mesure faite).
    """
    global _YAMLS
    if racine is not None:
        return sorted(racine.rglob("*.yaml"))
    if _YAMLS is None:
        _YAMLS = sorted(dossier_taches().rglob("*.yaml"))
    return _YAMLS


def _index_noms() -> dict[str, list[Path]]:
    """Index ``nom DECLARE -> YAML`` (``group`` et ``task``), bati une fois.

    Un seul parcours pour tout le repertoire : le refaire par tache faisait
    durer la resolution plus de quatre minutes (mesure faite).
    """
    global _INDEX_NOMS
    if _INDEX_NOMS is None:
        index: dict[str, list[Path]] = {}
        for yaml in _yamls():
            champs = _champs(yaml)
            for cle in ("group", "task"):
                nom = champs.get(cle)
                if nom:
                    index.setdefault(nom, []).append(yaml)
        _INDEX_NOMS = index
    return _INDEX_NOMS


def _candidats(tache: str, racine: Path | None) -> list[Path]:
    """YAML qui DECLARENT ``tache``, sinon qui s'appellent comme elle.

    Le nom de fichier ne suffit pas, et c'est un bug corrige : TROIS fichiers
    s'appellent ``_mmlu_stem.yaml`` (variantes llama3, llama3_cot, et le vrai
    mmlu). Le tri par chemin tombait sur la variante Llama, dont
    l'output_type vaut generate_until — donc « mmlu_stem » ressortait
    mesurable alors qu'il ne l'est pas. lm-eval resout par le nom DECLARE
    (``group:`` / ``task:``), pas par le nom de fichier : on fait pareil.
    Le nom declare passe donc AVANT l'homonymie de fichier.
    """
    if racine is not None:
        ys = _yamls(racine)
        declares = [y for y in ys if tache in (_champs(y).get("group"),
                                               _champs(y).get("task"))]
        fichiers = [y for y in ys if y.name in (f"{tache}.yaml", f"_{tache}.yaml")]
    else:
        declares = list(_index_noms().get(tache, []))
        fichiers = [y for y in _yamls()
                    if y.name in (f"{tache}.yaml", f"_{tache}.yaml")]
    return declares + [y for y in fichiers if y not in declares]


def _type_dans(chemin: Path, profondeur: int) -> str | None:
    """``output_type`` d'un YAML : la cle, sinon ``include``, sinon ses enfants."""
    if profondeur > _PROFONDEUR_MAX:
        return None
    champs = _champs(chemin)
    if champs.get("output_type"):
        return champs["output_type"]
    # `include:` : c'est le TEMPLATE qui porte l'output_type. Cas MMLU : la
    # tache mmlu_abstract_algebra.yaml ne declare qu'un dataset_name et un
    # include vers _default_template_yaml, dans le meme dossier.
    inclure = champs.get("include")
    if inclure:
        for nom in (inclure, inclure + ".yaml"):
            voisin = chemin.parent / nom
            if voisin.is_file():
                t = _type_dans(voisin, profondeur + 1)
                if t:
                    return t
    # Groupe : ses enfants se resolvent dans SON dossier. Deux raisons, et la
    # premiere est un bug corrige : 1) hors du dossier on tombe sur une
    # variante homonyme (flan_cot, generative) et « mmlu_stem » ressortait
    # generate_until au lieu de multiple_choice ; 2) cela evite de relire tout
    # le repertoire des taches a chaque niveau.
    for sous in _sous_taches(chemin):
        t = _type_par_nom(sous, profondeur + 1, chemin.parent)
        if t:
            return t
    return None


def _type_par_nom(tache: str, profondeur: int, racine: Path | None) -> str | None:
    """``output_type`` d'une tache nommee, cherchee sous ``racine``."""
    if profondeur > _PROFONDEUR_MAX:
        return None
    for yaml in _candidats(tache, racine):
        t = _type_dans(yaml, profondeur)
        if t:
            return t
    # ``tache`` peut etre une ETIQUETTE, pas un fichier : _mmlu_stem.yaml liste
    # « mmlu_stem_tasks », qui est un `tag` porte par les vraies taches du meme
    # dossier. On ne cherche QUE dans ce dossier — ailleurs on trouverait les
    # variantes flan/generative, qui notent autrement.
    if racine is None:
        return None
    for yaml in _yamls(racine):
        if _champs(yaml).get("tag") == tache:
            t = _type_dans(yaml, profondeur + 1)
            if t:
                return t
    return None


def type_sortie(tache: str) -> str | None:
    """``output_type`` declare par la tache dans les YAML de lm-eval.

    ``None`` si la tache est inconnue de cette installation. Les taches A
    CHOIX MULTIPLE (arc_easy) et les GROUPES (mmlu) se resolvent comme la
    tache elle-meme : c'est ce qui permet de les refuser, au lieu de
    produire un score de hasard.
    """
    return _type_par_nom(tache, 0, None)


def taches_loglikelihood(taches: str) -> list[str]:
    """Sous-ensemble de ``taches`` note par logprobs (donc non mesurable).

    Une tache inconnue de cette installation n'est pas refusee : on ne peut
    pas deviner son ``output_type``, on la laisse passer.
    """
    return [t.strip() for t in taches.split(",")
            if t.strip() and type_sortie(t.strip()) in _TYPES_LOGPROBS]


def construire_commande(taches: str, limite: int | None = None,
                        adresse_serveur: str | None = None,
                        modele: str = "aura-1b",
                        sortie: str | None = None,
                        brut: bool = False,
                        fewshot: int | None = None) -> list[str]:
    """Argv de ``python -m lm_eval`` pour noter le serveur local.

    Le transport est ``local-chat-completions`` (ou ``local-completions`` avec
    ``brut=True``) : aucun poids n'est recharge ici, ce qui evite un second
    modele en RAM. ``tokenizer_backend=None`` suffit pour les taches
    generatives et evite de trainer ``transformers``.
    """
    chat = not brut
    modele_lm = "local-chat-completions" if chat else "local-completions"
    args = [
        sys.executable, "-X", "utf8", "-m", "lm_eval",
        "--model", modele_lm,
        "--model_args",
        f"base_url={base_url(adresse_serveur, chat=chat)},model={modele},"
        "tokenizer_backend=None,num_concurrent=1",
        "--tasks", taches,
    ]
    if chat:
        args.append("--apply_chat_template")
    if fewshot is not None:
        args += ["--num_fewshot", str(fewshot)]
    if limite:
        args += ["--limit", str(limite)]
    if sortie:
        args += ["--output_path", sortie]
    return args


def lm_eval_installe() -> bool:
    """``lm_eval`` est-il importable ? (on ne l'importe qu'ici)."""
    import importlib.util

    return importlib.util.find_spec("lm_eval") is not None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lm_eval_aura")
    ap.add_argument("--taches", default=_TACHES_DEFAUT,
                    help="taches lm-eval, separees par des virgules")
    ap.add_argument("--limite", type=int, default=None,
                    help="nombre d'exemples par tache (rapide si petit)")
    ap.add_argument("--fewshot", type=int, default=None,
                    help="exemples en contexte (defaut : celui de la tache)")
    ap.add_argument("--modele", default="aura-1b",
                    help="nom annonce a lm-eval (cosmetique)")
    ap.add_argument("--sortie", default=None,
                    help="fichier/repertoire de resultats")
    ap.add_argument("--brut", action="store_true",
                    help="completion brute au lieu du gabarit de conversation")
    ap.add_argument("--loglikelihood", action="store_true",
                    help="forcer une tache a loglikelihood (scores NON valides "
                         "contre llama.cpp, cf. en-tete du module)")
    ap.add_argument("--liste", action="store_true",
                    help="lister les taches de lm-eval puis quitter")
    args = ap.parse_args(argv)

    if not lm_eval_installe():
        print("[lm-eval] non installe. Prerequis :\n"
              "    pip install lm-eval\n"
              "    (venv sans pip : uv pip install --python "
              ".venv/Scripts/python.exe lm-eval)", file=sys.stderr)
        return 2

    if args.liste:
        return subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "lm_eval", "--tasks", "list"],
            check=False, env=_ENV_UTF8).returncode

    refus = taches_loglikelihood(args.taches)
    if refus and not args.loglikelihood:
        print("[lm-eval] refus : tache(s) notee(s) par logprobs "
              + ", ".join(refus) + ".\n"
              "  llama.cpp n'echo pas les logprobs du PROMPT (issue #27174),\n"
              "  donc ces taches ne sont pas mesurables ici : soit une erreur,\n"
              "  soit un score de hasard silencieux. Utiliser une tache\n"
              "  generative (gsm8k, triviaqa...), ou --loglikelihood pour\n"
              "  forcer en connaissance de cause (chiffres NON valides).",
              file=sys.stderr)
        return 2
    if refus:
        print("[lm-eval] ATTENTION : " + ", ".join(refus)
              + " exige des logprobs de\n  prompt que llama.cpp ne fournit pas"
              " -> chiffres NON valides.", file=sys.stderr)

    cmd = construire_commande(args.taches, args.limite, modele=args.modele,
                              sortie=args.sortie, brut=args.brut,
                              fewshot=args.fewshot)
    print("[lm-eval] serveur :", base_url(chat=not args.brut))
    print("[lm-eval]", " ".join(cmd))
    return subprocess.run(cmd, check=False, env=_ENV_UTF8).returncode


if __name__ == "__main__":
    raise SystemExit(main())
