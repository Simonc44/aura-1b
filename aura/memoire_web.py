"""Recherche web avec extraits, provenance et Google Search Grounding optionnel.

La recherche reste locale/gratuite via DDGS par defaut. Si
``AURA_GEMINI_API_KEY`` est configuree, Gemini utilise son outil Google Search
et renvoie une synthese accompagnee des sources de grounding. Les pages et les
syntheses web restent des donnees non fiables pour le modele local.
"""
from __future__ import annotations

import logging
import os
import re

LOG = logging.getLogger("aura.web")

_MOTS_CLES_WEB = (
    "actualité", "actualite", "qui est", "qui est-ce", "quand", "dernier",
    "nouvelle", "découverte", "decouverte", "2024", "2025", "2026",
    "prix", "record", "match", "résultat", "resultat", "météo", "meteo",
)
_URL_RE = re.compile(r"https?://[^\s<>\]\[\)\}\|]+", re.IGNORECASE)


def a_besoin_web(question: str) -> bool:
    """Routeur : cette question a-t-elle besoin de faits externes ?"""
    q = question.lower()
    return any(mot in q for mot in _MOTS_CLES_WEB)


def _chercher_google(requete: str, timeout: int) -> str:
    """Interroge Gemini avec Google Search Grounding (cle fournie par l'user)."""
    cle = os.environ.get("AURA_GEMINI_API_KEY", "").strip()
    if not cle:
        return ""
    try:
        import requests

        timeout = int(os.environ.get("AURA_GEMINI_TIMEOUT", "20"))
        modele = os.environ.get("AURA_GEMINI_MODEL", "gemini-3.8-flash")
        url = ("https://generativelanguage.googleapis.com/v1beta/models/"
               f"{modele}:generateContent")
        reponse = requests.post(
            url,
            headers={"x-goog-api-key": cle,
                     "Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": (
                    "Réponds brièvement à la question avec Google Search. "
                    "Distingue les faits des incertitudes. Question : "
                    + requete)}]}],
                "tools": [{"google_search": {}}],
            },
            timeout=timeout,
        )
        reponse.raise_for_status()
        data = reponse.json()
        candidats = data.get("candidates") or []
        candidat = candidats[0] if candidats else {}
        parties = candidat.get("content", {}).get("parts", [])
        synthese = " ".join(str(p.get("text", "")).strip()
                            for p in parties if p.get("text")).strip()
        metadonnees = candidat.get("groundingMetadata", {})
        morceaux = []
        if synthese:
            morceaux.append(f"Synthèse de recherche Google : {synthese}")
        vus = set()
        for chunk in metadonnees.get("groundingChunks", []):
            web = chunk.get("web", {})
            titre = str(web.get("title", "Source Google")).strip()
            uri = str(web.get("uri", "")).strip()
            if uri.startswith("https://") and uri not in vus:
                vus.add(uri)
                morceaux.append(f"Titre : {titre} | Extrait : "
                                f"Source : {uri}")
        if not morceaux:
            LOG.info("Google grounding n'a renvoye aucune preuve exploitable")
            return ""
        return "\n".join(morceaux)
    except Exception as e:  # cle invalide, reseau ou quota -> repli DDGS
        LOG.info("Google grounding indisponible, repli DDGS (%s)", e)
        return ""


def chercher(requete: str, max_resultats: int = 3,
             timeout: int | None = None) -> str:
    """Renvoie un contexte avec titre, extrait et URL; vide si echec.

    Si ``AURA_GEMINI_API_KEY`` est present, essaie Google Search Grounding
    avant DDGS. La cle n'est jamais ajoutee a l'URL ni aux journaux.
    """
    if timeout is None:
        timeout = int(os.environ.get("AURA_WEB_TIMEOUT", "6"))
    google = _chercher_google(requete, timeout)
    if google:
        return google
    try:
        try:
            from ddgs import DDGS
        except ImportError:  # ancien nom de paquet
            from duckduckgo_search import DDGS  # type: ignore[no-redef]
    except ImportError:
        LOG.warning("paquet ddgs absent : memoire web desactivee")
        return ""

    morceaux = []
    try:
        with DDGS(timeout=timeout) as ddgs:
            for r in ddgs.text(requete, max_results=max_resultats):
                titre = (r.get("title") or "Source web").strip()
                extrait = (r.get("body") or "").strip()
                url = (r.get("href") or r.get("url") or "").strip()
                if titre or extrait:
                    morceaux.append(f"Titre : {titre} | Extrait : {extrait}"
                                    + (f" | Source : {url}" if url else ""))
    except Exception as e:  # reseau off, rate-limit... : jamais bloquant
        LOG.info("recherche web indisponible : %s", e)
        return ""
    return "\n".join(morceaux)


def extraire_sources(contexte: str, maximum: int = 5) -> list[tuple[str, str]]:
    """Extrait des liens HTTPS uniques (titre, URL) d'un contexte web."""
    sources: list[tuple[str, str]] = []
    vus: set[str] = set()
    for ligne in (contexte or "").splitlines():
        titre_match = re.search(r"Titre\s*:\s*(.*?)\s*\|", ligne,
                                re.IGNORECASE)
        titre = titre_match.group(1).strip() if titre_match else "Source web"
        for url in _URL_RE.findall(ligne):
            url = url.rstrip(".,;:")
            if url.startswith("https://") and url not in vus:
                vus.add(url)
                sources.append((titre, url))
                if len(sources) >= maximum:
                    return sources
    return sources


def ajouter_citations(reponse: str, contexte: str) -> str:
    """Ajoute une courte liste de liens des sources utilisees."""
    sources = extraire_sources(contexte)
    if not reponse or not sources:
        return reponse
    deja_citees = set(_URL_RE.findall(reponse))
    manquantes = [(titre, url) for titre, url in sources
                  if url not in deja_citees]
    if not manquantes:
        return reponse
    liens = "; ".join(f"[{titre}]({url})" for titre, url in manquantes)
    return f"{reponse.rstrip()}\n\nSources : {liens}"
