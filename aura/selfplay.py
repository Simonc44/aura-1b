"""Self-Play — boucle fermee d'auto-amelioration (l'esprit d'AlphaGo).

Le systeme joue contre lui-meme :
1. le GENERATEUR DE DEFIS fabrique des enigmes depuis le graphe de faits
   (attributions croisees : « A est lie a X, B a Y, donc qui... ? ») et
   des defis de code (fonctions pures avec reponse verifiable) ;
2. le RESOLVEUR (cerveau + experts symboliques) cherche la solution ;
3. le JUGE est deterministe (logique.py = backtracking exact, potcode =
   sandbox AST, calcul mental) — jamais le modele qui s'auto-note ;
4. succes -> le chemin gagnant est RENFORCE dans le graphe (MLT) et
   l'exemple est stocke (datasets/rlss.jsonl) ; echec -> correction
   auto-injectee.

Différence avec rlss.py (questions/réponses de savoir) : ici ce sont des
PROBLEMES a déduire, pas des faits a réciter. La recompense vient du
verificateur exact : une énigme résolue prouve un raisonnement valide.
"""
import itertools
import json
import logging
import os
import random
import time
from pathlib import Path

LOG = logging.getLogger("aura.selfplay")

_RACINE = Path(__file__).resolve().parent.parent
_SORTIE = _RACINE / "datasets" / "selfplay.jsonl"

# noms fictifs + lieux/objets pour construire des enigmes inedites
_NOMS = ("Aline", "Bruno", "Clara", "Dario", "Elena", "Farid")
_LIEUX = ("bibliotheque", "jardin", "marche", "piscine", "musee", "port")
_OBJETS = ("roman", "parapluie", "panier", "serviette", "carnet", "carte")

# sujets "generaux" du graphe qui servent de donnees pour les enigmes
_CATEGORIES_FAIT = ("capitale de", "president de")


def _faits_graphe(nb: int = 8) -> list[dict]:
    from . import graphe_faits
    try:
        stats = graphe_faits.stats()
    except Exception:
        return []
    triplets = []
    try:
        with open(graphe_faits._FICHIER, encoding="utf-8") as f:
            for ligne in f:
                try:
                    t = json.loads(ligne)
                except Exception:
                    continue
                if t.get("r") in _CATEGORIES_FAIT:
                    triplets.append(t)
                if len(triplets) >= nb:
                    break
    except OSError:
        return []
    return triplets


# ── Generateur de defis : enigmes d'attribution exactes ────────────────

def generer_enigme(nb_personnes: int = 3) -> dict | None:
    """Fabrique une enigme d'attribution + sa solution EXACTE.

    Chaque personne va a un lieu distinct et porte un objet distinct.
    L'enigme donne : les indices positifs (chaque personne a UN indice
    direct) + une contrainte croisee. La solution est generee d'abord
    (permutation aleatoire) : la validation par logique.py est donc
    exacte par construction.
    """
    noms = random.sample(_NOMS, nb_personnes)
    lieux = random.sample(_LIEUX, nb_personnes)
    objets = random.sample(_OBJETS, nb_personnes)

    qui_va_ou = dict(zip(noms, lieux))
    qui_prend_quoi = dict(zip(noms, objets))

    # indices : pour chaque personne, « X va au lieu » OU « X porte objet »
    lignes = []
    moitie = max(1, nb_personnes // 2)
    for i, nom in enumerate(noms):
        if i < moitie:
            article = "a la " if lieux[i][0] in "bpm" else "au "
            lignes.append(f"{nom} va {article}{lieux[i]}.")
        else:
            article = "l'" if objets[i][0] in "aeiou" else "le/la "
            lignes.append(f"{nom} porte {article}{objets[i]}.")
    # contrainte croisee : le lieu de la personne 0 n'est pas celui de la 1
    lignes.append(f"Chaque personne va a un endroit different et porte "
                  f"un objet different.")
    # question : ou va la derniere personne ?
    cible = noms[-1]
    question = (f"Enigme logique : {' '.join(lignes)} Ou va {cible} ? "
                f"Reponds uniquement par le nom de l'endroit.")
    attendu = qui_va_ou[cible]
    return {"type": "logique", "question": question, "attendu": attendu,
            "faits": qui_va_ou, "cible": cible}


def generer_defi_calcul() -> dict:
    """Defi de calcul mental a reponse exacte (validation triviale)."""
    a, b = random.randint(12, 39), random.randint(7, 29)
    if random.random() < 0.5:
        question = f"combien fait {a} fois {b} ? reponds juste le nombre"
        attendu = str(a * b)
    else:
        question = f"combien fait {a * b + b} divise par {b} ? reponds juste le nombre"
        attendu = str(a + 1)
    return {"type": "calcul", "question": question, "attendu": attendu}


def generer_defi(niveau: int = 1) -> dict:
    """Un defi au hasard : enigme (niveau >= 2) ou calcul (niveau 1)."""
    if niveau >= 2 and random.random() < 0.4:
        enigme = generer_enigme(nb_personnes=random.choice((3, 3, 4)))
        return enigme if enigme else generer_defi_calcul()
    return generer_defi_calcul()


# ── Juge deterministe ──────────────────────────────────────────────────

def _juger(reponse: str, attendu: str) -> bool:
    """Validation exacte (normalisation tolérante) — jamais le modele."""
    if not reponse:
        return False
    basse = reponse.lower()
    return attendu.lower() in basse and len(basse) <= 200


# ── Boucle de self-play ────────────────────────────────────────────────

def session(nb_defis: int | None = None, niveau: int = 1) -> dict:
    """Joue N defis : generer -> resoudre -> juger -> integrer.

    Succes : renforcement des faits du graphe impliques + exemple JSONL.
    Echec : correction auto-injectee (prochaine fois, le contexte la
    montrera). AURA_SELFPLAY=N pour lancer en ligne de commande.
    """
    nb_defis = nb_defis or int(os.environ.get("AURA_SELFPLAY", "0") or 0) or 10
    from . import llama_cerveau, graphe_faits

    _SORTIE.parent.mkdir(parents=True, exist_ok=True)
    succes = echecs = 0
    t0 = time.time()
    faits = _faits_graphe()

    for _ in range(nb_defis):
        defi = generer_defi(niveau=niveau)
        if defi is None:
            continue
        reponse = llama_cerveau.generer(defi["question"], categorie="math")
        ok = _juger(reponse, defi["attendu"])
        if ok:
            succes += 1
            exemple = {"user": defi["question"],
                       "assistant": reponse.strip()[:200],
                       "type": defi["type"]}
            with _SORTIE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(exemple, ensure_ascii=False) + "\n")
            # REINTEGRATION : les faits du graphe utilises par l'enigme
            # sont renforces (schema de pensee eprouve -> MLT)
            if defi["type"] == "logique" and faits:
                paire = random.choice(faits)
                try:
                    graphe_faits.renforcer(paire["s"], paire["o"])
                except Exception:
                    pass
        else:
            echecs += 1
            try:
                from . import autoamelioration
                autoamelioration.enregistrer_correction(
                    defi["question"], reponse.strip()[:120],
                    defi["attendu"], "math")
            except Exception:
                pass

    bilan = {"defis": succes + echecs, "succes": succes, "echecs": echecs,
             "taux": round(succes / max(succes + echecs, 1), 2),
             "secondes": round(time.time() - t0, 1)}
    LOG.info("[selfplay] session : %s", bilan)
    return bilan


if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.INFO)
    print(json.dumps(session(), ensure_ascii=False, indent=2))
