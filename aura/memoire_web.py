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
import unicodedata

LOG = logging.getLogger("aura.web")

# BRIDAGE DU RAG (fix bug 1, 12 oct. 2026) : limite materielle stricte sur
# le nombre de morceaux (chunks) injectes dans le prompt.
# - RAG_MAX_DEFAUT : 3 morceaux pour les requetes classiques (suffisant,
#   evite le bruit sur CPU). Configurable via AURA_RAG_MAX.
# - RAG_MAX_ABSOLU : plafond absolu de 10 morceaux, quoi qu'il arrive.
#   Plus jamais de prompts inondes a 154 morceaux qui etouffent le CPU
#   et provoquent des fuites de metadonnees.
RAG_MAX_ABSOLU = 10  # jamais plus, meme pour les grandes recherches


def _entier_borne(valeur: str, defaut: int, bas: int, haut: int) -> int:
    try:
        return max(bas, min(haut, int(valeur)))
    except (TypeError, ValueError):
        return defaut


# AURA_RAG_MAX est BORNE : une valeur folle (154, 0, "abc") ne peut ni
# rouvrir la porte a l'inondation ni vider le contexte.
RAG_MAX_DEFAUT = _entier_borne(os.environ.get("AURA_RAG_MAX", "3"), 3, 1,
                               RAG_MAX_ABSOLU)

# Recherche COMPLEXE (comparaison, dossier, synthese multi-sources) : seule
# situation qui autorise plus que RAG_MAX_DEFAUT, jamais plus que le plafond.
_RE_RECHERCHE_COMPLEXE = re.compile(
    r"\b(compar\w+|synthese|dossier|rapport|etat des lieux|panorama|"
    r"recherche approfondie|en detail|detaille\w*|exhaustif\w*|"
    r"plusieurs sources|differentes sources|avantages et inconvenients|"
    r"points de vue|chronologie|historique complet)\b")
_LONGUEUR_QUESTION_COMPLEXE = 300


def recherche_complexe(question: str) -> bool:
    """Vrai si la demande justifie plus de contexte que le defaut."""
    q = _normaliser(question)
    return bool(_RE_RECHERCHE_COMPLEXE.search(q)
                or len(q) >= _LONGUEUR_QUESTION_COMPLEXE)


def limite_morceaux(question: str = "") -> int:
    """Nombre MAXIMAL de morceaux injectables : 3 (classique) ou 10 (complexe)."""
    return RAG_MAX_ABSOLU if recherche_complexe(question) else RAG_MAX_DEFAUT


def limiter_morceaux(contexte: str, maximum: int | None = None,
                     question: str = "") -> str:
    """Coupe un contexte a `maximum` morceaux (lignes non vides, doublons
    retires, ordre conserve). Dernier verrou materiel avant le prompt : meme
    si un etage amont deborde, jamais plus de RAG_MAX_ABSOLU morceaux.
    """
    if not contexte:
        return contexte
    if maximum is None:
        maximum = limite_morceaux(question)
    maximum = max(1, min(int(maximum), RAG_MAX_ABSOLU))
    lignes = list(dict.fromkeys(ligne.rstrip() for ligne in contexte.splitlines()
                                if ligne.strip()))
    if len(lignes) <= maximum:
        return contexte
    LOG.info("[rag] contexte borne : %d -> %d morceaux", len(lignes), maximum)
    return "\n".join(lignes[:maximum])

_MOTS_CLES_WEB = (
    "actualité", "actualite", "qui est", "qui est-ce", "quand", "dernier",
    "nouvelle", "découverte", "decouverte", "2024", "2025", "2026",
    "prix", "record", "match", "résultat", "resultat", "météo", "meteo",
    "aujourd'hui", "en ce moment", "actuel", "actuelle", "récemment",
    "recemment", "dernières nouvelles", "dernieres nouvelles",
    "sur le web", "sur internet", "en ligne", "recherche web",
    "cherche sur internet", "cherche sur le web", "recherche sur internet",
    "recherche sur le web", "sources web", "search the web",
    "search online", "look up online", "on the web",
)
_MOTS_DEMANDE_WEB = (
    "sur le web", "sur internet", "en ligne", "recherche web",
    "cherche sur internet", "cherche sur le web", "recherche sur internet",
    "recherche sur le web", "sources web", "search the web",
    "search online", "look up online", "on the web",
)
_URL_RE = re.compile(r"https?://[^\s<>\]\[\)\}\|]+", re.IGNORECASE)


def a_besoin_web(question: str) -> bool:
    """Routeur : cette question a-t-elle besoin de faits externes ?"""
    q = _normaliser(question)
    return any(_normaliser(mot) in q for mot in _MOTS_CLES_WEB)


def demande_explicite_web(question: str) -> bool:
    """Vrai si la personne demande explicitement une recherche Internet."""
    q = _normaliser(question)
    return any(_normaliser(mot) in q for mot in _MOTS_DEMANDE_WEB)


def _normaliser(texte: str) -> str:
    texte = unicodedata.normalize("NFKD", (texte or "").lower())
    return "".join(c for c in texte if not unicodedata.combining(c))


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


def chercher_iteratif(question: str, cible: str = "",
                      max_resultats: int | None = None,
                      timeout: int | None = None, tours: int = 2) -> str:
    """Recherche une preuve en deux passes au plus, la seconde ciblant la cible.

    Le score est un filtre de pertinence lexical, pas une preuve de veracite.
    Les resultats restent a verifier par le modele avant de reviser sa reponse.

    BRIDAGE (fix bug 1) : max_resultats est plafonne a RAG_MAX_ABSOLU (10),
    et le reranker final est limite a RAG_MAX_DEFAUT (3) par defaut.
    """
    def mots(texte: str) -> set[str]:
        texte = unicodedata.normalize("NFKD", texte.lower())
        texte = "".join(c for c in texte if not unicodedata.combining(c))
        return {m for m in re.findall(r"[a-z0-9]{3,}", texte)
                if m not in {"pour", "avec", "dans", "quel", "quelle",
                             "estce", "comment", "what", "where", "when",
                             "the", "and", "for", "was", "are"}}

    # BRIDAGE STRICT : 3 morceaux pour une requete classique, 10 au plus pour
    # une recherche complexe. `plafond` s'impose meme a un appelant qui
    # demande davantage.
    plafond = limite_morceaux(question)
    max_resultats = min(max_resultats or plafond, plafond)
    requetes = [question.strip()]
    cible_courte = " ".join(cible.strip().split()[:18])
    if cible_courte:
        requetes.append(f'{question.strip()} "{cible_courte}" source officielle')
    else:
        # Deuxième passe de secours : apporte un biais vers des sources
        # documentées si les premiers extraits sont rares ou trop génériques.
        requetes.append(f"{question.strip()} source fiable")
    contextes: list[str] = []
    urls: set[str] = set()
    for requete in requetes[:max(1, min(int(tours), 2))]:
        try:
            resultat = chercher(requete, max_resultats=max_resultats,
                                timeout=timeout)
        except Exception as exc:
            LOG.info("recherche iterative indisponible: %s", exc)
            continue
        lignes = []
        termes_question = mots(question)
        termes_cible = mots(cible)
        for ligne in resultat.splitlines():
            termes_ligne = mots(ligne)
            pertinent = (len(termes_ligne & termes_question) >= 1
                         or bool(termes_ligne & termes_cible))
            if not pertinent:
                continue
            urls_ligne = set(_URL_RE.findall(ligne))
            if urls_ligne and urls_ligne <= urls:
                continue
            urls.update(urls_ligne)
            lignes.append(ligne)
        if not lignes and resultat.strip():
            # Un extrait pertinent peut utiliser des synonymes absents de la
            # requête. Garder les premiers candidats permet au reranker
            # sémantique de les sauver au lieu de perdre toute la recherche.
            lignes = [ligne.strip() for ligne in resultat.splitlines()
                      if ligne.strip()]
        if lignes:
            contextes.extend(lignes)
            # Une source directement pertinente suffit à éviter un appel reseau.
            if (cible_courte and any(mots(ligne) & termes_cible
                                     for ligne in lignes)):
                break
            if not cible_courte and len(contextes) >= 3:
                break
    # BRIDAGE FINAL : meme apres deux passes (et leurs lignes cumulees), on
    # ne renvoie jamais plus que le plafond de la requete (3 ou 10).
    return _reranker_preuves(question, cible, contextes, maximum=plafond)


def _reranker_preuves(question: str, cible: str,
                      lignes: list[str], maximum: int = 3) -> str:
    """Garde les meilleures lignes avant injection au petit modele.

    MiniLM partage l'encodeur ONNX deja utilise ailleurs par RontoGPT. S'il
    manque, une similarite lexicale sans dependance prend le relais.

    BRIDAGE (fix bug 1) : le parametre maximum est plafonne a RAG_MAX_ABSOLU
    pour empecher toute injection de plus de 10 morceaux, meme si l'appelant
    passe une valeur superieure.
    """
    # APPLICATION DU BRIDAGE : jamais plus de RAG_MAX_ABSOLU morceaux
    maximum = min(maximum, RAG_MAX_ABSOLU)
    candidates = list(dict.fromkeys(ligne.strip() for ligne in lignes
                                    if ligne.strip()))
    if not candidates:
        return ""

    scores: list[float] | None = None
    try:
        from . import embeddings

        textes = [question.strip()]
        if cible.strip():
            textes.append(cible.strip()[:500])
        vecteurs = embeddings.encoder(textes + candidates)
        if vecteurs is not None and len(vecteurs) == len(textes) + len(candidates):
            # encodeur() renvoie des vecteurs L2-normalises : le produit
            # scalaire est le cosinus, calcule en un seul produit matriciel.
            sims = vecteurs[:len(textes)] @ vecteurs[len(textes):].T
            scores = (0.8 * sims[0] + 0.2 * sims[1]
                      if len(textes) > 1 else sims[0]).tolist()
    except Exception as exc:  # le RAG ne depend jamais de MiniLM
        LOG.debug("reranking semantique indisponible: %s", exc)

    if scores is None:
        def mots(texte: str) -> set[str]:
            texte = unicodedata.normalize("NFKD", texte.lower())
            texte = "".join(c for c in texte if not unicodedata.combining(c))
            return set(re.findall(r"[a-z0-9]{3,}", texte))

        q = mots(question)
        c = mots(cible)
        scores = [len((q | c) & mots(ligne)) / max(len(q | c), 1)
                  for ligne in candidates]

    meilleurs = sorted(range(len(candidates)), key=lambda i: scores[i],
                       reverse=True)[:max(1, maximum)]
    return "\n".join(candidates[i] for i in meilleurs)


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
