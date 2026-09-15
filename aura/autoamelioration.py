"""Systeme d'auto-amélioration : apprend de chaque feedback utilisateur.

Principe simple (inspiré du cerveau humain) :
- quand l'utilisateur dit "c'est mal" ou donne une correction, on stocke
  la question + la correction dans un fichier JSONL
- a la prochaine question similaire, on injecte les corrections comme contexte
  supplementaire pour le LLM (pas de fine-tuning, juste du RAG sur les erreurs)

C'est le meme principe que la memoire a court terme -> memoire a long terme :
les corrections les plus recemment utiles sont priorisées.
"""
import json
import os
import time
from pathlib import Path

import logging

LOG = logging.getLogger("aura.autoamelioration")

# fichier de stockage des corrections
_FICHIER = os.path.join(os.path.dirname(__file__), ".cache_corrections.jsonl")


def _charger_corrections() -> list[dict]:
    """Charge toutes les corrections depuis le fichier."""
    if not os.path.exists(_FICHIER):
        return []
    corrections = []
    try:
        with open(_FICHIER, encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                if ligne:
                    corrections.append(json.loads(ligne))
    except Exception as e:
        LOG.warning("lecture corrections impossible : %s", e)
    return corrections


def enregistrer_correction(question: str, reponse_faute: str, correction: str,
                           categorie: str = "general") -> None:
    """Stocke une correction pour futur apprentissage."""
    entree = {
        "question": question,
        "reponse_faute": reponse_faute,
        "correction": correction,
        "categorie": categorie,
        "timestamp": time.time(),
    }
    try:
        with open(_FICHIER, "a", encoding="utf-8") as f:
            f.write(json.dumps(entree, ensure_ascii=False) + "\n")
        LOG.info("[auto] correction enregistree pour: %s", question[:50])
    except Exception as e:
        LOG.warning("ecriture correction impossible : %s", e)


def chercher_corrections_similaires(question: str, max_resultats: int = 3) -> list[str]:
    """Cherche des corrections applicables a une question donnee.

    Methode simple : similarité par mots en commun (pas de ML).
    """
    corrections = _charger_corrections()
    if not corrections:
        return []

    q_mots = set(question.lower().split())
    scores = []
    for c in corrections:
        c_mots = set(c.get("question", "").lower().split())
        # score = nb de mots en commun / nb mots question
        intersection = q_mots & c_mots
        if intersection:
            score = len(intersection) / max(len(q_mots), 1)
            scores.append((score, c))

    # trier par score, prendre les meilleurs
    scores.sort(key=lambda x: x[0], reverse=True)
    resultats = []
    for score, c in scores[:max_resultats]:
        if score > 0.2:  # seuil minimum
            resultats.append(
                f"[correction anterieure] Question similaire : "
                f"\"{c['question'][:80]}\" → reponse correcte : {c['correction']}"
            )
    return resultats


def construire_contexte_corrections(question: str) -> str:
    """Construit un bloc de texte a injecter dans le prompt du LLM."""
    corrections = chercher_corrections_similaires(question)
    if not corrections:
        return ""
    return "\n".join(corrections) + "\n\n"


def stats() -> dict:
    """Retourne les statistiques d'apprentissage."""
    corrections = _charger_corrections()
    categories = {}
    for c in corrections:
        cat = c.get("categorie", "general")
        categories[cat] = categories.get(cat, 0) + 1
    return {"total": len(corrections), "par_categorie": categories}
