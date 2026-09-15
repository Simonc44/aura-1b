"""Pilier memoire : recherche web DuckDuckGo (le web = disque dur externe).

Paquet officiel renomme `duckduckgo_search` -> `ddgs` : on accepte les deux.
Aucune cle API, aucune limite locale. Utilise pour l'actualite et la culture
generale, ce qui libere les parametres du modele pour le langage et la logique.
"""
import logging

LOG = logging.getLogger("aura.web")

_MOTS_CLES_WEB = (
    "actualité", "actualite", "qui est", "qui est-ce", "quand", "dernier",
    "nouvelle", "découverte", "decouverte", "2024", "2025", "2026",
    "prix", "record", "match", "résultat", "resultat", "météo", "meteo",
)


def a_besoin_web(question: str) -> bool:
    """Routeur : cette question a-t-elle besoin de faits externes ?"""
    q = question.lower()
    return any(mot in q for mot in _MOTS_CLES_WEB)


def chercher(requete: str, max_resultats: int = 3, timeout: int = 10) -> str:
    """Renvoie un contexte texte filtre (titre + extrait), vide si echec."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:  # ancien nom de paquet
            from duckduckgo_search import DDGS
    except ImportError:
        LOG.warning("paquet ddgs absent : memoire web desactivee")
        return ""

    morceaux = []
    try:
        with DDGS(timeout=timeout) as ddgs:
            for r in ddgs.text(requete, max_results=max_resultats):
                titre = (r.get("title") or "").strip()
                extrait = (r.get("body") or "").strip()
                if titre or extrait:
                    morceaux.append(f"Titre : {titre} | Extrait : {extrait}")
    except Exception as e:  # reseau off, rate-limit... : jamais bloquant
        LOG.info("recherche web indisponible : %s", e)
        return ""
    return "\n".join(morceaux)
