"""RLSS — Renforcement Symbolique par Succes (auto-apprentissage en boucle).

Principe : le systeme GENERE ses propres exercices depuis le graphe de
faits (aura/.graphe_faits.jsonl), y REPOND avec le cerveau, VALIDE par
les experts deterministes (filtre instantane/PAL, graphe) et ne conserve
que les reponses justes comme exemples d'entrainement. Le signal de
recompense est SYMBOLIQUE (verification exacte), jamais un modele qui
s'auto-note — c'est le meme contrat que potcode (le code rate ne sort
jamais) etendu au savoir.

Boucle complete (AURA_RLSS=1 pour lancer une session) :
1. tirer N triplets verifies du graphe de faits ;
2. fabriquer une question + la reponse attendue (deterministe) ;
3. demander la reponse au cerveau (chemin serveur si actif) ;
4. valider : la reponse contient-elle l'attendu (tolerant casse/accents) ?
5. succes -> exemple JSONL dans datasets/rlss.jsonl (formulaire
   question/reponse, pret pour le prochain LoRA Colab) ;
   echec -> entree dans .cache_corrections.jsonl (le prompt d'auto-
   amelioration existant la ressortira a la prochaine question proche).

Capacite : session = AURA_RLSS questions (defaut 20). Aucun reseau
externe, aucun GPU : quelques secondes par question.
"""
import json
import logging
import os
import random
import re
import time
import unicodedata
from pathlib import Path

LOG = logging.getLogger("aura.rlss")

_RACINE = Path(__file__).resolve().parent.parent
_GRAPHE = _RACINE / "aura" / ".graphe_faits.jsonl"
_SORTIE = _RACINE / "datasets" / "rlss.jsonl"
_CORRECTIONS = _RACINE / ".cache_corrections.jsonl"

# modeles de questions (categorie, gabarit) — la reponse attendue est
# TOUJOURS l'objet du triplet : la validation est exacte par construction
_GABARITS = (
    ("general", "Que sais-tu sur {sujet} ? Reponds en une phrase courte."),
    ("general", "Qui ou quoi est {sujet} ? Reponds en une phrase courte."),
    ("web", "Donne un fait precis a propos de {sujet}."),
)


def _normaliser(texte: str) -> str:
    """Minuscules, sans accents ni ponctuation (comparaison tolerant)."""
    t = unicodedata.normalize("NFD", texte.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _triplets() -> list[dict]:
    """Les triplets verifies du graphe (sujet, relation, objet)."""
    if not _GRAPHE.is_file():
        return []
    triplets = []
    try:
        for ligne in _GRAPHE.read_text(encoding="utf-8").splitlines():
            ligne = ligne.strip()
            if not ligne:
                continue
            try:
                t = json.loads(ligne)
            except Exception:
                continue
            if isinstance(t, dict):
                sujet = t.get("sujet") or t.get("s")
                objet = t.get("objet") or t.get("o")
                if sujet and objet:
                    triplets.append({"sujet": str(sujet), "objet": str(objet)})
    except Exception as e:
        LOG.info("[rlss] graphe illisible : %s", e)
    return triplets


def _fabriquer_exercice(triplet: dict) -> tuple[str, str, str] | None:
    """(categorie, question, attendu) depuis un triplet — None si inutilisable."""
    sujet = str(triplet.get("sujet", "")).strip()
    objet = str(triplet.get("objet", "")).strip()
    if len(sujet) < 3 or len(objet) < 2:
        return None
    gabarit = random.choice(_GABARITS)
    question = gabarit[1].format(sujet=sujet)
    return gabarit[0], question, objet


def _valider(reponse: str, attendu: str) -> bool:
    """La reponse cite-t-elle l'objet attendu ? (normalisation tolerant)"""
    if not reponse:
        return False
    attendu_n = _normaliser(attendu)
    if len(attendu_n) < 2:
        return False
    # reponse courte (<= 12 mots) : egalite stricte acceptable ; sinon
    # inclusion (l'objet doit apparaitre dans la reponse)
    return attendu_n in _normaliser(reponse)


def session(nb: int | None = None) -> dict:
    """Execute une session RLSS complete, renvoie le bilan."""
    nb = nb or int(os.environ.get("AURA_RLSS", "0") or 0) or 20
    triplets = _triplets()
    if not triplets:
        LOG.info("[rlss] graphe vide -> session annulee (nourris-le avec "
                 "scripts/nourrir_graphe.py)")
        return {"exercices": 0, "succes": 0, "echecs": 0, "dataset": 0}

    random.shuffle(triplets)
    from . import llama_cerveau

    _SORTIE.parent.mkdir(parents=True, exist_ok=True)
    succes = echecs = 0
    t0 = time.time()
    for triplet in triplets[:nb]:
        exercice = _fabriquer_exercice(triplet)
        if exercice is None:
            continue
        categorie, question, attendu = exercice
        reponse = llama_cerveau.generer(question, categorie=categorie)
        if _valider(reponse, attendu):
            succes += 1
            with _SORTIE.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"user": question,
                                    "assistant": reponse.strip()},
                                   ensure_ascii=False) + "\n")
        else:
            echecs += 1
            # l'echec devient une correction auto-injectee (mechanisme
            # existant d'autoamelioration) : la prochaine question proche
            # verra la bonne reponse
            try:
                from . import autoamelioration
                autoamelioration.enregistrer_correction(
                    question, reponse.strip()[:120], attendu, categorie)
            except Exception:
                pass

    bilan = {"exercices": succes + echecs, "succes": succes, "echecs": echecs,
             "dataset": _SORTIE.stat().st_size // 1024 if _SORTIE.is_file() else 0,
             "secondes": round(time.time() - t0, 1)}
    LOG.info("[rlss] session : %s", bilan)
    return bilan


if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.INFO)
    print(json.dumps(session(), ensure_ascii=False, indent=2))
