"""Cache de reponses en memoire (question -> resultat), extrait de RontoGPT.

Prudent par construction : jamais de question contextuelle ou courte (elle
depend de la conversation), jamais de refus ni d'erreur, TTL court pour le
web (les faits bougent). AURA_CACHE_REP_TTL=0 le coupe.
"""
import os
import re
import time

from . import filtre_instantane
from .detection_demande import _sans_accents
from .regles_reponse import _RE_SUITE_CONVERSATION, _est_refus

_TTL_REPONSE_S = float(os.environ.get("AURA_CACHE_REP_TTL", "900"))
_TTL_REPONSE_WEB_S = min(_TTL_REPONSE_S, 300.0)
_CACHE_REPONSES_MAX = 128
_CACHE_REPONSES: dict[str, tuple[float, dict]] = {}


def cle_reponse(question: str, X, y) -> str | None:
    """Cle de cache normalisee, ou None si la question n'est pas cachable."""
    if X or y or _TTL_REPONSE_S <= 0:
        return None
    brut = question.lower()
    if any(m in brut for m in filtre_instantane._MOTS_CONTEXTUELS):
        return None
    q = re.sub(r"[^a-z0-9' ]+", " ", _sans_accents(question).lower())
    q = " ".join(q.split())
    if len(q.split()) < 4 or _RE_SUITE_CONVERSATION.match(q):
        return None
    return q


def lire(cle: str | None) -> dict | None:
    """Resultat memorise encore valide pour cette cle, sinon None."""
    if cle is None:
        return None
    vu = _CACHE_REPONSES.get(cle)
    if vu is None:
        return None
    expire, resultat = vu
    if time.time() >= expire:
        _CACHE_REPONSES.pop(cle, None)
        return None
    return dict(resultat)


def memoriser(cle: str | None, resultat: dict) -> None:
    """Memorise un resultat cachable ; ignore les reponses volatiles."""
    if cle is None or not isinstance(resultat, dict):
        return
    rep = str(resultat.get("reponse", ""))
    methodes = (resultat.get("analyse") or {}).get("methodes") or []
    if (not rep.strip() or rep.startswith("[RontoGPT]") or _est_refus(rep)
            or resultat.get("clarification") or resultat.get("en_attente")
            or resultat.get("cerveau_choisi") == "logique-non-formalisee"
            or "temps" in methodes):          # heure : volatile, jamais cachee
        return
    ttl = (_TTL_REPONSE_WEB_S if "web" in (resultat.get("experts") or ())
           else _TTL_REPONSE_S)
    if len(_CACHE_REPONSES) >= _CACHE_REPONSES_MAX:
        _CACHE_REPONSES.pop(next(iter(_CACHE_REPONSES)), None)
    _CACHE_REPONSES[cle] = (time.time() + ttl, dict(resultat))
