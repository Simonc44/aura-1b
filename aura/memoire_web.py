"""Pilier memoire : recherche web DuckDuckGo (le web = disque dur externe).

Paquet officiel renomme `duckduckgo_search` -> `ddgs` : on accepte les deux.
Aucune cle API, aucune limite locale. Utilise pour l'actualite et la culture
generale, ce qui libere les parametres du modele pour le langage et la logique.

Recherche enrichie (mode riche) : double requete (faits + analyses) et
extraction d'un lexique de « briques de langage » injectees au modele pour
ameliorer la redaction.
"""
import logging
import re
from collections import Counter

LOG = logging.getLogger("aura.web")

_MOTS_CLES_WEB = (
    "actualité", "actualite", "qui est", "qui est-ce", "quand", "dernier",
    "nouvelle", "découverte", "decouverte", "2024", "2025", "2026",
    "prix", "record", "match", "résultat", "resultat", "météo", "meteo",
)

# stopwords FR : exclus du lexique (mots outils, pas du jargon)
_STOPWORDS = frozenset(
    "apres donc alors cela cette ces dans avec pour plus moins tout tous toute "
    "toutes aussi entre devant chez depuispendant lorsque parce leur leurs notre "
    "votre nos vos etre avoir fait faire peut peuvent seront etait etaient comme "
    "mais donc ainsi selon pendant encore toujours jamais souvent enfin assez "
    "trop tres moins autre autres meme memes premier premiere deuxieme dernier "
    "derniere jusqu'a jusqu'a afin quand alors puisque quel quelle quels quelles "
    "notre votre etc bien pierre paul Marie".split()
)

_MOTS = re.compile(r"[a-zàâäéèêëïîôöùûüç]{6,}")


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


def _extraire_lexique(textes: str, max_mots: int = 8) -> list[str]:
    """Isole les « briques de langage » : mots rares/techniques des articles.

    Frequence + longueur >= 6, stopwords exclus. Ces termes orientent le
    vocabulaire du modele vers le jargon correct du domaine.
    """
    freq = Counter(
        m for m in _MOTS.findall(textes.lower()) if m not in _STOPWORDS)
    return [mot for mot, _ in freq.most_common(max_mots)]


def chercher_enrichi(question: str, max_resultats: int = 3) -> str:
    """Mode riche : double requete + lexique injecte dans le contexte.

    - requete 1 : les FAITS (question brute)
    - requete 2 : le STYLE / les analyses (question + analyse synthese essai)
    - lexique   : jargon extrait des deux resultats

    Chaque bloc est etiquete ; le prompt de style s'appuie sur ces sections.
    """
    faits = chercher(question, max_resultats=max_resultats)
    analyses = chercher(f"{question} analyse synthese essai",
                        max_resultats=2)
    if not faits and not analyses:
        return ""

    lexique = _extraire_lexique(f"{faits}\n{analyses}")
    parties = []
    if faits:
        parties.append(f"FAITS WEB (temps reel) :\n{faits}")
    if analyses:
        parties.append(f"ANALYSES (points de vue) :\n{analyses}")
    if lexique:
        parties.append("Mots-cles pertinents a utiliser : ["
                       + ", ".join(lexique) + "]")
    return "\n\n".join(parties)
